## Latest changes
* Added new Traffic Scenarios
    - PassingFromOppositeDirections: Scenario used to calibrating and testing sensor models: ego vehicle passes other vehicle from opposite direction on a long straight.
    - OvertakingSlowTarget: Scenario used for testing a simple overtake maneuver. Ego car should overtake other vehicle with safe margin on a long straight.
    - FollowingAcceleratingTarget: Scenario used for testing whether ego car follows a slowly accelerating other vehicle correctly on a long straight.
    - FollowingDeceleratingTarget: Scenario used for testing whether ego car follows a slowly decelerating other vehicle correctly on a long straight.
    - FollowingChangingLanesTarget: Scenario used for testing a cut off situation where other vehicle is slowly changing lane to one occupied by ego.
    - DrivingOffDriveway: Scenario used for testing a situation where other vehicle is merging from covered driveway right in front of ego.
    - OncomingTargetDriftsOntoEgoLane: Scenario where oncoming other vehicle is slowly drifting from its lane onto one occupied by ego.
* Added new behaviors
    - DriveVehicleContinuous: Controlled vehicle will have the specified control values applied to itself with optional max_speed parameter to limit acceleration once desired speed is achieved.
    - DriveToLocationContinuous: Similar to DriveVehicleContinuous but with lateral control so that vehicle will point to target_location
    - FollowVehicleContinuous: Similar to DriveVehicleContinuous but with lateral control so that vehicle will steer towards target_vehicle
    - TriggerOnLocation: check if a vehicle is within certain distance to a target location
    - TriggerOnStatusChange: check if a vehicle status value(s) have passed target value(s) (x, y, z, roll, pitch, yaw)
* Updated folder structure and naming convention in lowecase
* Reworked scenario execution
    - Every scenario has to have a configuration provided as XML file.
      Currently there is one XML file for each scenario class
    - The scenario runner is now responsible for spawning/destroying all actors.
    - Renamed vehicle -> actor
    - If all scenarios in one coniguration file should be executed, the scenario_runner can be started with --scenario group:<CONFIG_FILE>
    - Generalized ControlLoss and FollowLeadingVehicle scenarios
    - Added randomization option to scenario_runner and scenarios
* Added new atomic behaviors using py_trees behavior tree library
    - BasicAgentBehavior: drive to target location using CARLA's BasicAgent
    - StandStill: check if a vehicle stands still
    - InTriggerDistanceToNextIntersection: check if a vehicle is within certain distance with respect to the next intersection
* Fixes
    - Fixed SteerVehicle atomic behavior to keep vehicle velocity

## CARLA Scenario_Runner 0.9.2

* Added Traffic Scenarios engine to reproduce complex traffic situations for training and evaluating driving agents
* Added NHTSA Traffic Scenarios
    - FollowLeadingVehicle: hero vehicle must react to the deccelerations of a leading vehicle
    - FollowLeadingVehicleWithObstacle: hero vehicle must react to a leading vehicle due to an obstacle blocking the road
    - StationaryObjectCrossing: hero vehicle must react to a cyclist or pedestrian blocking the road
    - DynamicObjectCrossing: hero vehicle must react to a cyclist or pedestrian suddenly crossing in front of it
    - OppositeVehicleRunningRedLight: hero vehicle must avoid a collision at an intersection regulated by traffic lights when the crossing traffic runs a red light
    - NoSignalJunctionCrossing: hero vehicle must cross a non-signalized intersection
    - VehicleTurningRight: hero vehicle must react to a cyclist or pedestrian crossing ahead after a right turn
    - VehicleTurningLeft: hero vehicle must react to a cyclist or pedestrian crossing ahead after a left turn
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
