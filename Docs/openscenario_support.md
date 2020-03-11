## OpenSCENARIO Support

The scenario_runner provides support for the [OpenSCENARIO](http://www.openscenario.org/) 1.0 standard.
The current implementation covers initial support for maneuver Actions, Conditions, Stories and the Storyboard.
If you would like to use evaluation criteria for a scenario to evaluate pass/fail results, these can be implemented
as StopTriggers (see below). However, not all features for these elements are yet available. If in doubt, please see the
module documentation in srunner/tools/openscenario_parser.py

An example for a supported scenario based on OpenSCENARIO is available [here](../srunner/examples/FollowLeadingVehicle.xosc)

In addition, it is recommended to take a look into the official documentation available [here](https://releases.asam.net/OpenSCENARIO/1.0.0/Model-Documentation/index.html) and [here](https://releases.asam.net/OpenSCENARIO/1.0.0/ASAM_OpenSCENARIO_BS-1-2_User-Guide_V1-0-0.html#_foreword).

### Migrating OpenSCENARIO 0.9.x to 1.0
The easiest way to convert old OpenSCENARIO samples to the official standard 1.0 is to use _xsltproc_ and the migration scheme located in the openscenario folder.
Example:

```bash
xsltproc -o newScenario.xosc migration0_9_1to1_0.xslt oldScenario.xosc
```


### Level of support
In the following the OpenSCENARIO attributes are listed with their current support status.

#### General OpenSCENARIO setup

This covers all part that are defined outside the OpenSCENARIO Storyboard

|Attribute  | Support Status | Notes / Remarks |
|-----------|:-------------------------:|-----------------|
|FileHeader | Yes | Use "CARLA:" at the beginning of the description to use the CARLA coordinate system |
|ParameterDeclarations | Yes | Parameters can currently only be defined at the beginning (i.e. globally) |
|CatalogLocations - VehicleCatalog    | Yes | |
|CatalogLocations - PedestrianCatalog | Yes | |
|CatalogLocations - MiscObjectCatalog | Yes | |
|CatalogLocations - EnvironmentCatalog| Yes | |
|CatalogLocations - ManeuverCatalog   | Yes | |
|CatalogLocations - ControllerCatalog | Yes | While the catalog is supported, the reference / usage may not work |
|CatalogLocations - TrajectoryCatalog | Yes | While the catalog is supported, the reference / usage may not work |
|CatalogLocations - RouteCatalog      | Yes | While the catalog is supported, the reference / usage may not work |
|RoadNetwork - LogicFile      | Yes | The CARLA level can be used directly (e.g. LogicFile=Town01). Also any OpenDRIVE path can be provided. |
|RoadNetwork - SceneGraphFile | No  | The provided information is not used |
|RoadNetwork - TrafficSignals | No  | The provided information is not used |
|Entities - EntitySelection   | No  | The provided information is not used |
|Entities - ScenarioObject - ObjectController | No  | The provided information is not used |
|Entities - ScenarioObject - CatalogReference | Yes | |
|Entities - ScenarioObject - Vehicle          | Yes | The name should match a CARLA vehicle model, otherwise a default vehicle based on the vehicleCategory is used. The color can be set via properties ('Property name="color" value="0,0,255"'). Axles, Performance, BoundingBox entries are ignored. |
|Entities - ScenarioObject - Pedestrian       | Yes | The name should match a CARLA vehicle model, otherwise a default vehicle based on the vehicleCategory is used. BoundingBox entries are ignored. |
|Entities - ScenarioObject - MiscObject       | Yes | The name should match a CARLA vehicle model, otherwise a default vehicle based on the vehicleCategory is used. BoundingBox entries are ignored. |

#### OpenSCENARIO Storyboard

##### OpenSCENARIO Actions

The OpenSCENARIO Actions can be used for two different purposes. First, Actions can be used to
define the initial behavior of something, e.g. a traffic participant. Therefore, Actions can be
used within the OpenSCENARIO Init. In addition, Actions are also used within the OpenSCENARIO
story. In the following, the support status for both application areas is listed. If an action
contains of submodules, which are not listed, the support status applies to all submodules.

###### GlobalAction

|Action  | Support within Init | Support within Story | Notes / Remarks |
|--------|:-------------------:|:--------------------:|-----------------|
|EnvironmentAction  | Yes | No | |
|EntityAction       | No  | No | |
|ParameterAction    | No  | No | |
|InfrastructureAction - TrafficSignalAction - TrafficSignalControllerAction  | No | No | |
|InfrastructureAction - TrafficSignalAction - TrafficSignalStateAction  | No | Yes | Setting a traffic light state in CARLA works by providing the position of the relevant traffic light (Example: TrafficSignalStateAction name="pos=x,y" state="green") |
|TrafficAction  | No | No | |

###### UserDefinedAction

|Action  | Support within Init | Support within Story | Notes / Remarks |
|--------|:-------------------:|:--------------------:|-----------------|
|CustomCommandAction  | No | Yes* | This action is currently used to trigger the execution of an additional script. Example: type="python /path/to/script args" |

###### PrivateAction

|Action  | Support within Init | Support within Story | Notes / Remarks |
|--------|:-------------------:|:--------------------:|-----------------|
|LongitudinalAction - SpeedAction                | Yes | Yes | |
|LongitudinalAction - LongitudinalDistanceAction | No  | Yes | |
|LateralAction - LaneChangeAction      | No | Yes* | Currently only lane change by one lane to the left or right is supported (RelativeTargetLane) |
|LateralAction - LaneOffsetAction      | No | No | |
|LateralAction - LateralDistanceAction | No | No | |
|VisibilityAction         | No  | No | |
|SynchronizeAction        | No  | No | |
|ActivateControllerAction | No  | Yes* | Only supports the autopilot at the moment |
|ControllerAction         | No  | No  | |
|TeleportAction           | Yes | Yes | |
|RoutingAction - AssignRouteAction      | No | Yes | |
|RoutingAction - FollowTrajectoryAction | No | No  | |
|RoutingAction - AcquirePositionAction  | No | No  | |


##### OpenSCENARIO Conditions

Conditions in OpenSCENARIO can be defined either as ByEntityCondition or as ByValueCondition.
Both can be used for StartTrigger and StopTrigger conditions.
The following two tables list the support status for each.

###### ByEntityCondition

|Condition  | Support Status | Notes / Remarks |
|-----------|:--------------:|-----------------|
|EndOfRoadCondition       | No  | |
|CollisionCondition       | Yes | |
|OffroadCondition         | No  | |
|TimeHeadwayCondition     | No  | |
|TimeToCollisionCondition | Yes | |
|AccelerationCondition    | No  | |
|StandStillCondition      | Yes | |
|SpeedCondition           | Yes | |
|RelativeSpeedCondition   | No  | |
|TraveledDistanceCondition| Yes | |
|ReachPositionCondition   | Yes | |
|DistanceCondition        | Yes | |
|RelativeDistanceCondition| Yes | |

###### ByValueCondition

|Condition  | Support Status | Notes / Remarks |
|-----------|:--------------:|-----------------|
|ParameterCondition               | Yes*| The level of support depends on the parameter. It is recommended to use other conditions if possible. Please also consider the note below. |
|TimeOfDayCondition               | No  | |
|SimulationTimeCondition          | Yes | |
|StoryboardElementStateCondition  | Yes*| startTransition, stopTransition, endTransition and completeState are currently supported|
|UserDefinedValueCondition        | No  | |
|TrafficSignalCondition           | No  | |
|TrafficSignalControllerCondition | No  | |

!!! Note
     In the OpenSCENARIO 1.0 standard, a definition of test / evaluation criteria is not
     defined. For this purpose, you can re-use StopTrigger conditions with CARLA. The following
     StopTrigger conditions for evaluation criteria are supported through ParameterConditions by
     providing the criteria name for the condition:

     * criteria_RunningStopTest
     * criteria_RunningRedLightTest
     * criteria_WrongLaneTest
     * criteria_OnSideWalkTest
     * criteria_KeepLaneTest
     * criteria_CollisionTest
     * criteria_DrivenDistanceTest

##### OpenSCENARIO Positions

There are several ways of defining positions in OpenSCENARIO. In the following we list the
current support status for each definition format.

|Position  | Support Status | Notes / Remarks |
|----------|:--------------:|-----------------|
|WorldPosition          | Yes | |
|RelativeWorldPosition  | Yes | |
|RelativeObjectPosition | Yes | |
|RoadPosition           | No  | |
|RelativeRoadPosition   | No  | |
|LanePosition           | Yes | |
|RelativeLanePosition   | Yes | |
|RoutePosition          | No  | |
