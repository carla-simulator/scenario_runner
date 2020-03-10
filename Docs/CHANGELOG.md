## Table of Contents
* [Latest Changes](#latest-changes)
* [CARLA ScenarioRunner 0.9.7](#carla-scenariorunner-097)
* [CARLA ScenarioRunner 0.9.6](#carla-scenariorunner-096)
* [CARLA ScenarioRunner 0.9.5.1](#carla-scenariorunner-0951)
* [CARLA ScenarioRunner 0.9.5](#carla-scenariorunner-095)
* [CARLA ScenarioRunner 0.9.2](#carla-scenariorunner-092)

## Latest Changes

## CARLA ScenarioRunner 0.9.8
### :rocket: New Features
* Added "--timeout" command line parameter to set a user-defined timeout value
* Scenario updates:
    - Changed traffic light behavior of scenarios 7, 8 and 9. The new sequence is meant to greatly improve the chances of the ego vehicle having to interact at junctions.
* OpenSCENARIO support:
    - Add initial support for Catalogs (Vehicle, Pedestrian, Environment, Maneuver, and and MiscObject types only)
### :bug: Bug Fixes
* Fixed #471: Handling of weather parameter (cloudyness -> cloudiness adaption)
* Fixed #472: Spawning issue of pedestrians in OpenSCENARIO
* Fixed #374: Usage of evaluation critieria with multiple ego vehicles in OpenSCENARIO
* Fixed #459: Add initial support for Catalogs (Vehicle, Pedestrian, Environment, Maneuver, and and MiscObject types only)
* Fixed wrong StandStill behavior which return SUCCESS immediatly on a standing actor
* Fixed scenario bug causing junction related scenarios (4, 7, 8 and 9) to not spawn due to lane changes.
### :ghost: Maintenance
* Added watchdog to ScenarioManager to handle timeouts and CARLA crashes
* Added timeout for CARLA tick() calls to avoid blocking CARLA server calls


## CARLA ScenarioRunner 0.9.7
**This is the _first_ release to work with CARLA 0.9.7 (not the patch versions 0.9.7.x)**
### :rocket: New Features
* Challenge routes can be directly executed with the ScenarioRunner using the --route option
* Agents can be used with the ScenarioRunner (currently only for route-based scenarios)
* New scenarios:
    - Added example scenario for lane change
    - Added cut-in example scenario
* Scenario updates:
    - Scenarios 7 to 10 are now visible when running routes (instead of being triggered in the background). Their
      methodology has remained unchanged
* Scenario atomics:
    - Added new OutsideRouteLanesTest atomic criter that encompasses both SidewalkTest and WrongLaneTest, returning
      the percentage of route that has been traveled outside the lane.
    - InRouteTest is now more forgiving. The max distance has been increased, but staying above the previous one will eventually 
      also cause failure
    - Changed SidewalkTest atomic criteria to also track other type of out of lane conditions
    - SidewalkTest and WrongLaneTest atomic criterias now track the amount of meters traversed
    - CollisionTest atomic criteria now correctly ignores multiple micro-collisions with the same object
    - Added LaneChange and TrafficLightSateSetter behavior atomics
    - Added AccelerateToCatchUp behavior atomic
    - Added get_transform() method for CarlaDataProvider
    - Added support for weather conditions
    - Added basic version check to ensure usage of correct CARLA version
    - WaypointFollower atomic can handle pedestrians
    - Extensions in WaypointFollower atomic for consecutive WaypointFollowers (one WF cancels the previous one)
* Extended OpenScenario support:
    - Added support for UserDefinedActions (e.g. to run additional scripts)
    - Added init speed behavior for vehicles
    - Added support for relative velocities
    - Extended convert_position_to_transform with RelativeWorld, RelativeObject and RelativeLane osc_positions
    - Added new trigger atomics InTriggerDistanceToOSCPosition and InTimeToArrivalToOSCPosition to support relative osc_positions
    - Added new atomic behaviour ActorTransformSetterToOSCPosition
    - Workaround for relative osc_positions: World is started earlier to support relative osc_positions in story init
    - Added delay condition support in convert_condition_to_atomic
    - Added support for pedestrians
    - Full support for SimulationTime condition
    - Added weather support
    - Updated implementation to be closer to upcoming OpenSCENARIO standard
    - AfterTermination, AtStart conditions are supported
    - Added initial support for lateral action: LaneChange
    - Added initial support for OSCGlobalAction to set state of traffic signal
    - FollowRoute action is supported for vehicles and pedestrians, for global world positions.
    - Added support for RoadCondition: Friction
    - Redundant rolename object property is no longer required
    - Added support for global parameters
    - Fixed coordinate system to use right-hand as default. Left-hand CARLA system can be used by adding "CARLA:" at the start of the description in the FileHeader.
    - Added support to change actor color
    - Added support for a default actor model, in case the stated model is not available
    - Added support for MiscObjects (besides vehicles and pedestrians)
    - Reworked traffic signal handling: The name has to start now either with "id=" or "pos=" depending on whether the position or id is used as unique identifier
    - Actor physics can now be set via Object Properties (<Property name="physics" value="off" />)
### :bug: Bug Fixes
* Fixed wrong handling of OpenSCENARIO ConditionGroups, which should be handled as parallel composites, not sequences
* Fixed #443: Repetitions in OpenSCENARIO were not properly working
* Fixed bug causing RunningStopTest atomic criteria to trigger when lane changing near a STOP signal
* Fixed bug causing RunningRedLightTest atomic criteria to occasionally not trigger
* Fixed bug causing occasional frame_errors
* Fixed #426: Avoid underground vehicles fall forever by disabling physics when spawning underground.
* Fixed #427: Removed unnecessary warnings when using get_next_traffic_light() with non-cached locations
* Fixed missing ego_vehicle: compare actor IDs instead of object in CarlaDataProvider in get_velocity, get_transform and get_location
* Avoided use of 'controller.ai.walker' as walker type in DynamicObjectCrossing scenario
* Fixed WaypointFollower behavior to use m/s instead of km/h
* Fixed starting position of VehicleTurnLeft/Right scenarios
* Fixed spawn_point modification inside CarlaActorPool.setup_actor()
* Fixed result of DrivenDistanceTest
* Fixed exception in manual_control on fps visualization
* Cleanup of pylint errors for all autonomous agents
* Fixed randomness of route-based scenarios
* Fixed usage of radians instead of degrees for OpenSCENARIO
* Fixed ActorTransformSetter behavior to avoid vehicles not reaching the desired transform
* Fixed spawning of debris for ControlLoss scenario (Scenario01)
* Fixed CTRL+C termination of ScenarioRunner
### :ghost: Maintenance
* Increased speed of actor initialization by using CARLA batch mode and buffering CARLA blueprint library
* Split of behaviors into behaviors and conditions
* Moved atomics into new submodule scenarioatomics
* Updated documentation for all behaviors, conditions and test criteria
* Refactoring of scenario configurations and parsers
* Extended WaypointFollower atomic behavior to be able to use the current actor speed
* Removed usage of 'import *' to have cleaner Python imports
* Removed broad-except and bare-except where possible
* Python-Scenarios: Removed obsolete categories
* ScenarioRunner: Removed scenario dictonary, use imports directly
* CarlaDataProvider: Simplified update_light_states() to remove code duplication
* Timer: class TimeOut() is derived from SimulationTimeCondition() to  avoid code duplication
* Moved backported py_trees classes and methods to tools/py_trees_port.py to avoid code duplication
* Removed setup_environment.sh
* Adaptions to CARLA API Changes
     - Renamed GnssEvent to GnssMeasurement

## CARLA ScenarioRunner 0.9.6
**This is the _first_ release to work with CARLA 0.9.6**
### :ghost: Maintenance
* Adapted to CARLA API changes
    - Frame rate is set now via Python
    - Renamed frame_count and frame_number to frame
    - Removed wait_for_tick() calls


## CARLA ScenarioRunner 0.9.5.1
**This is the _last_ release that works with CARLA 0.9.5**
### :rocket: New Features
* Added initial support for OpenScenario v0.9.1
* Added support for multiple ego vehicles plus an example
* Added commandline option for output directory
* Added option to load external scenario implementations (in python)
* Added option to scenario_runner to load external scenario XMLs
* Atomic behaviors:
    - Extended KeepVelocity atomic behavior to support duration/distance
      based termination
    - Extended StandStill atomic behavior to support duration based
      termination
    - Added behavior to activate/deactivate autopilot
### :bug: Bug Fixes
* Fixed WaypointFollower initialization


## CARLA ScenarioRunner 0.9.5
**This is the _first_ release to work with CARLA 0.9.5**
### :rocket: New Features
* Added support for CARLA challenge
    - Added logging functionalities to challenge_evaluator_routes.py
    - Added wall clock timeout for the CARLA challenge
    - Added background scenario to generate dynamic traffic using autopilot
    - Updated compatibility with Python 2.7 for the challenge evaluator
    - Updated WaypointFollower behavior
    - Added detect_lane_obstacle() helper function which identifies if an obstacle is present in front of the reference actor
    - Added test to detect vehicles running a stop
    - Updated the reference position for a scenario is now called trigger_point
    - Added universal access to the map without re-calling get_map()
    - Added criteria_enable flag to enable/disable criteria tree
    - Added multiple helper methods for generic scenario execution.
    - Added pseudo-sensors for SceneLayoutMeasurements and ObjectMeasurements for Track4 of the CARLA AD challenge
    - Added track identification for autonomous_agent.py
    - Added HDMap pseudo-sensor
    - Added new traffic event logger
    - Added various helper methods to allow generic scenario execution
    - Added method to calculate distance along a route
    - In challenge mode spawn exception are caught and the corresponding scenario is removed
* Added new atomic behaviors using py_trees behavior tree library
    - BasicAgentBehavior: drive to target location using CARLA's BasicAgent
    - StandStill: check if a vehicle stands still
    - InTriggerDistanceToNextIntersection: check if a vehicle is within certain distance with respect to the next intersection
    - WaypointFollower: follows auto-generated waypoints indefinitely or follows a given waypoint list
    - HandBrakeVehicle: sets the handbrake value for a given actor
    - ActorDestroy: destroys a given actor
    - ActorTransformSetter: sets transform of given actor
    - ActorSource: creates actors indefinitely around a location if no other vehicles are present within a threshold
    - ActorSink: indefinitely destroys vehicles that wander close to a location within a threshold
    - InTriggerDistanceToLocationAlongRoute: check if an actor is within a certain distance to a given location along a given route
* Added new atomic evaluation criteria
    - Added running red light test
    - Added running stop test
    - Added wrong way test
* Added NHTSA Traffic Scenarios
    - Updated all traffic scenarios to let the other actors appear upon scenario triggering and removal on scenario end
    - ManeuverOppositeDirection: hero vehicle must maneuver in the opposite lane to pass a leading vehicle.
    - OtherLeadingVehicle: hero vehicle must react to the deceleration of leading vehicle and change lane to avoid collision and follow the vehicle in changed lane
    - SignalizedJunctionRightTurn: hero vehicle must turn right into the same direction of another vehicle crossing straight initially from a lateral direction and avoid collision at a signalized intersection.
    - SignalizedJunctionLeftTurn : hero vehicle is turning left at signalized intersection, cuts across the path of another vehicle coming straight crossing from an opposite direction.
### :bug: Bug Fixes
* Fixed SteerVehicle atomic behavior to keep vehicle velocity    
### :ghost: Maintenance
* Reworked scenario execution
    - Updated folder structure and naming convention in lowercase
    - Extended CarlaDataProvider with method to get next relevant traffic light
    - Every scenario has to have a configuration provided as XML file.
      Currently there is one XML file for each scenario class
    - The scenario runner is now responsible for spawning/destroying the ego vehicle
    - Added a CarlaActorPool to share scenario-related actors between scenarios and the scenario_runner
    - Renamed vehicle -> actor
    - If all scenarios in one configurations file should be executed, the scenario_runner can be started with --scenario group:<CONFIG_FILE>
    - Generalized ControlLoss and FollowLeadingVehicle scenarios
    - Added randomization option to scenario_runner and scenarios
    - The scenario behavior always starts with a wait behavior until the ego vehicle reached the scenario starting position
    - Created method _initialize_actors in basic scenario that can be overridden for scenario specific actor initialization
* Updated NHTSA Traffic Scenarios
    - OppositeVehicleRunningRedLight: Updated to allow execution at different locations    


## CARLA ScenarioRunner 0.9.2
**This release is designed to work with CARLA 0.9.2**
### :rocket: New Features
* Added Traffic Scenarios engine to reproduce complex traffic situations for training and evaluating driving agents
* Added NHTSA Traffic Scenarios
    - FollowLeadingVehicle: hero vehicle must react to the deceleration of a leading vehicle
    - FollowLeadingVehicleWithObstacle: hero vehicle must react to a leading vehicle due to an obstacle blocking the road
    - StationaryObjectCrossing: hero vehicle must react to a cyclist or pedestrian blocking the road
    - DynamicObjectCrossing: hero vehicle must react to a cyclist or pedestrian suddenly crossing in front of it
    - OppositeVehicleRunningRedLight: hero vehicle must avoid a collision at an intersection regulated by traffic lights when the crossing traffic runs a red light
    - NoSignalJunctionCrossing: hero vehicle must cross a non-signalized intersection
    - VehicleTurningRight: hero vehicle must react to a cyclist or pedestrian crossing ahead after a right turn
    - VehicleTurningLeft: hero vehicle must react to a cyclist or pedestrian crossing ahead after a left turn
    - ControlLoss: Hero vehicle must react to a control loss and regain its control
* Added atomic behaviors using py_trees behavior trees library
    - InTriggerRegion: new behavior to check if an object is within a trigger region
    - InTriggerDistanceToVehicle: check if a vehicle is within certain distance with respect to a reference vehicle
    - InTriggerDistanceToLocation: check if a vehicle is within certain distance with respect to a reference location
    - TriggerVelocity: triggers if a velocity is met
    - InTimeToArrivalToLocation:  check if a vehicle arrives within a given time budget to a reference location
    - InTimeToArrivalToVehicle: check if a vehicle arrives within a given time budget to a reference vehicle
    - AccelerateToVelocity: accelerate until reaching requested velocity
    - KeepVelocity: keep constant velocity
    - DriveDistance: drive certain distance
    - UseAutoPilot: enable autopilot
    - StopVehicle: stop vehicle
    - WaitForTrafficLightState: wait for the traffic light to have a given state
    - SyncArrival: sync the arrival of two vehicles to a given target
    - AddNoiseToVehicle: Add noise to steer as well as throttle of the vehicle
