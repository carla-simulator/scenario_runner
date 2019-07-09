## Latest changes

## CARLA Scenario_Runner 0.9.5.1
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
    - Fixed WaypointFollower initialization


## CARLA Scenario_Runner 0.9.5
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
* Fixes
    - Fixed SteerVehicle atomic behavior to keep vehicle velocity
* Updated NHTSA Traffic Scenarios
    - OppositeVehicleRunningRedLight: Updated to allow execution at different locations
* Added NHTSA Traffic Scenarios
    - Updated all traffic scenarios to let the other actors appear upon scenario triggering and removal on scenario end
    - ManeuverOppositeDirection: hero vehicle must maneuver in the opposite lane to pass a leading vehicle.
    - OtherLeadingVehicle: hero vehicle must react to the deceleration of leading vehicle and change lane to avoid collision and follow
                           the vehicle in changed lane
    - SignalizedJunctionRightTurn: hero vehicle must turn right into the same direction of another vehicle crossing
                                   straight initially from a lateral direction and avoid collision at a signalized intersection.
    - SignalizedJunctionLeftTurn : hero vehicle is turning left at signalized intersection, cuts across the path of another vehicle
                                   coming straight crossing from an opposite direction.

## CARLA Scenario_Runner 0.9.2

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
