# List of scenarios

ScenarioRunner ships scenarios of two kinds, both implemented as `BasicScenario` subclasses in `srunner/scenarios/`:

*   **Standalone scenarios** have a configuration in `srunner/examples/*.xml` and are run by name with `--scenario`. There are __19 types__ in __96 configurations__ on this branch, spread over `Town01_Opt` to `Town05_Opt` and `Town10HD_Opt`. `--list` prints the names.
*   **Route scenarios** are triggered along a route, at a trigger point in the route file, and run with `--route`. There are __26 types__ that appear only in the route files under `srunner/data/`. Most place their actors relative to the trigger point, so they work anywhere on any route.

Seven standalone types are not used by any route, and the rest of the standalone types double as route scenarios.

---
## Standalone scenarios

Run with `--scenario <name>`, where the name is a configuration name such as `FollowLeadingVehicle_1`.

| Type | Configs | Towns | What happens |
|---|---|---|---|
| `FollowLeadingVehicle` | 11 | Town01–05 | A lead vehicle drives down the road, slows and stops. The ego has to stop behind it. |
| `FollowLeadingVehicleWithObstacle` | 11 | Town01–05 | As above, with an obstacle hidden in front of the lead vehicle. |
| `OtherLeadingVehicle` | 10 | Town04, Town05 | Two other vehicles ahead; the lead one slows down and the ego has to react or change lane. |
| `DynamicObjectCrossing` | 9 | Town02–05 | A cyclist or pedestrian comes out from behind a blocker and crosses in front of the ego. |
| `StationaryObjectCrossing` | 8 | Town02–05 | A cyclist stands still in the middle of the lane, blocking the ego. |
| `VehicleTurningRight` | 8 | Town01–04 | The ego turns right at a junction and a cyclist crosses its path. |
| `VehicleTurningLeft` | 8 | Town01–04 | The same, turning left. |
| `SignalizedJunctionRightTurn` | 7 | Town03–05 | The ego turns right at a signalized junction while another vehicle crosses. |
| `OppositeVehicleRunningRedLight` | 6 | Town03, Town04, Town10HD | Another vehicle runs its red light across the ego's path at a signalized junction. |
| `ConstructionObstacle` | 4 | Town02, Town04, Town10HD | A construction setup blocks the lane and the ego has to change lane. |
| `ManeuverOppositeDirection` | 4 | Town03–05 | The ego has to overtake using the opposite lane, with oncoming traffic. |
| `BlockedIntersection` | 2 | Town10HD | The ego turns and finds the exit blocked by a stopped vehicle. |
| `ParkingExit` | 2 | Town10HD | The ego starts in a parking slot, between two vehicles, and has to merge into the driving lane. |
| `Accident` | 1 | Town04 | An accident with a police vehicle blocks the lane, forcing a lane change. |
| `ControlLoss` | 1 | Town10HD | Noise is injected into the ego's control and friction patches appear under it. |
| `EnterActorFlow` | 1 | Town04 | The ego has to merge into a continuous flow of vehicles. |
| `SignalizedJunctionLeftTurn` | 1 | Town10HD | The ego turns left at a signalized junction across oncoming traffic. |
| `StaticCutIn` | 1 | Town10HD | A vehicle parked on a side lane pulls out and cuts in front of the ego. |
| `VehicleOpensDoorTwoWays` | 1 | Town10HD | A parked vehicle opens its door, forcing the ego into the opposite lane. |

---
## Route scenarios

These types appear in the route files in `srunner/data/` and are triggered as the ego reaches their trigger point. They have no standalone configuration.

| Type | What happens |
|---|---|
| `AccidentTwoWays` | `Accident`, with the ego forced to invade the opposite lane. |
| `ConstructionObstacleTwoWays` | `ConstructionObstacle`, invading the opposite lane. |
| `ParkedObstacle` | A badly parked vehicle forces the ego out of its lane. |
| `ParkedObstacleTwoWays` | The same, invading the opposite lane. |
| `HazardAtSideLane` | Bicycles ride on the side of a road with no sidewalk. |
| `HazardAtSideLaneTwoWays` | The same, invading the opposite lane. |
| `VehicleTurningRoute` | `VehicleTurning` with the adversary placed from the route path. |
| `VehicleTurningRoutePedestrian` | The same, with a pedestrian instead of a cyclist. |
| `ParkingCrossingPedestrian` | `DynamicObjectCrossing` where the blocker is a parked vehicle. |
| `PedestrianCrossing` | A group of pedestrians crosses the road ahead. |
| `ParkingCutIn` | A parked vehicle cuts in front of the ego, forcing it to brake. |
| `HighwayCutIn` | A vehicle merges into the ego's lane from an entry lane. |
| `HighwayExit` | The ego has to leave the highway across a flow of vehicles. |
| `EnterActorFlowV2` | `EnterActorFlow` for highway entries with dedicated lanes. |
| `MergerIntoSlowTraffic` | The ego merges into slow-moving traffic. |
| `MergerIntoSlowTrafficV2` | A variation of the above. |
| `InterurbanActorFlow` | The ego leaves an interurban road by turning left across a flow. |
| `InterurbanAdvancedActorFlow` | The ego joins an interurban road, crossing a flow and merging into another. |
| `CrossingBicycleFlow` | The ego crosses a flow of bicycles at a junction. |
| `InvadingTurn` | An oncoming vehicle invades the ego's lane on a turn. |
| `NonSignalizedJunctionLeftTurn` | `SignalizedJunctionLeftTurn` without traffic lights. |
| `NonSignalizedJunctionRightTurn` | `SignalizedJunctionRightTurn` without traffic lights. |
| `OppositeVehicleTakingPriority` | An oncoming vehicle takes priority at an unsignalized junction. |
| `YieldToEmergencyVehicle` | An emergency vehicle approaches from behind and the ego has to make way. |
| `HardBreakRoute` | The background traffic in front of the ego brakes hard. |
| `PriorityAtJunction` | Sets the ego's next traffic light green, so it can gather speed. A support scenario. |

Several of these drive the surrounding traffic through the background activity rather than spawning their own actors, using the commands in `srunner/tools/background_manager.py`.

---
## Where they come from

Many of the standalone scenarios implement situations from the NHTSA pre-crash typology, and the class docstrings name the traffic scenario number where there is one — for example `ControlLoss` is Traffic Scenario 01 and `FollowLeadingVehicle` is Traffic Scenario 02.

To add your own, see [Create a new scenario](creating_new_scenario.md). Nothing has to be registered: a `BasicScenario` subclass whose name matches a configuration's `type` is found automatically, including from a folder outside the checkout passed with `--additionalScenario`.
